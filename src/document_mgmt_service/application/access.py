from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from document_mgmt_service.application.metadata import redact_sensitive_text
from document_mgmt_service.application.search import SemanticSearchService
from document_mgmt_service.domain.models import (
    DocumentPrivacyClassification,
    DocumentSummaryRecord,
    DocumentVersionRecord,
    SemanticContentSearchResult,
    SemanticDocumentSearchResult,
    SemanticEvidenceResult,
)
from document_mgmt_service.domain.ports import FileStorage, PostgreSQLDocumentRepository


class AccessAction(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    REDACT = "REDACT"


class AccessIntent(str, Enum):
    METADATA = "METADATA"
    DESCRIPTION = "DESCRIPTION"
    PAGE = "PAGE"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
    CONTENT_SEARCH = "CONTENT_SEARCH"
    EVIDENCE = "EVIDENCE"
    WHOLE_DOCUMENT = "WHOLE_DOCUMENT"


_SENSITIVE_QUERY_PATTERNS = [
    r"\bssn\b",
    r"\bsocial security\b",
    r"\baadhaar\b",
    r"\bpassport\b",
    r"\bcredit card\b",
    r"\bbank account\b",
    r"\baccount number\b",
    r"\bpassword\b",
    r"\bsecret\b",
    r"\btoken\b",
    r"\bapi key\b",
    r"\bmedical\b",
    r"\bdiagnosis\b",
    r"\bprescription\b",
    r"\bsalary\b",
    r"\bphone\b",
    r"\bemail\b",
]
_SENSITIVE_TEXT_RE = re.compile("|".join(_SENSITIVE_QUERY_PATTERNS), re.IGNORECASE)
_PII_RE = re.compile(
    r"\b(?:\d{3}-\d{2}-\d{4}|\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\+?\d[\d\s().-]{7,}\d)\b"
)


@dataclass(frozen=True, slots=True)
class AccessRequest:
    document_id: str
    version: int
    intent: AccessIntent
    query: str | None = None
    limit: int = 10
    approval_granted: bool = False
    requestor: str | None = None


@dataclass(frozen=True, slots=True)
class AccessDecision:
    action: AccessAction
    reason: str
    approval_required: bool = False


@dataclass(frozen=True, slots=True)
class AccessAuditEvent:
    timestamp: datetime
    request: AccessRequest
    decision: AccessDecision
    document_privacy: DocumentPrivacyClassification
    redacted: bool = False


class AccessDeniedError(RuntimeError):
    def __init__(self, decision: AccessDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class ApprovalRequiredError(RuntimeError):
    def __init__(self, decision: AccessDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class DocumentAccessPolicy:
    def evaluate(
        self,
        *,
        record: DocumentVersionRecord,
        request: AccessRequest,
    ) -> AccessDecision:
        if request.intent in {AccessIntent.METADATA, AccessIntent.DESCRIPTION}:
            return AccessDecision(AccessAction.ALLOW, "metadata and description are low-risk")

        if request.intent == AccessIntent.WHOLE_DOCUMENT:
            return self._evaluate_whole_document(record, request)

        if request.intent in {AccessIntent.PAGE, AccessIntent.CONTENT_SEARCH, AccessIntent.EVIDENCE, AccessIntent.DOCUMENT_SEARCH}:
            return self._evaluate_content_access(record, request)

        return AccessDecision(AccessAction.DENY, "unsupported access intent", approval_required=False)

    def _evaluate_whole_document(self, record: DocumentVersionRecord, request: AccessRequest) -> AccessDecision:
        if record.privacy == DocumentPrivacyClassification.OPEN:
            return AccessDecision(AccessAction.ALLOW, "open documents may be retrieved in full")
        if record.privacy == DocumentPrivacyClassification.SENSITIVE:
            return AccessDecision(AccessAction.REQUIRE_APPROVAL, "sensitive documents require approval before full retrieval", approval_required=True)
        if record.privacy in {DocumentPrivacyClassification.OPEN_NOT_PUBLIC, DocumentPrivacyClassification.PRIVATE}:
            if request.approval_granted:
                return AccessDecision(AccessAction.ALLOW, "approval granted for full-document retrieval")
            return AccessDecision(AccessAction.REQUIRE_APPROVAL, "full-document retrieval requires approval", approval_required=True)
        return AccessDecision(AccessAction.REQUIRE_APPROVAL, "full-document retrieval requires approval", approval_required=True)

    def _evaluate_content_access(self, record: DocumentVersionRecord, request: AccessRequest) -> AccessDecision:
        query = request.query or ""
        targeted_sensitive = self._is_targeted_sensitive_query(query)

        if record.privacy == DocumentPrivacyClassification.OPEN:
            if targeted_sensitive:
                return AccessDecision(AccessAction.REQUIRE_APPROVAL, "targeted sensitive query requires approval", approval_required=True)
            return AccessDecision(AccessAction.ALLOW, "open content may be searched")

        if targeted_sensitive:
            return AccessDecision(AccessAction.REQUIRE_APPROVAL, "targeted sensitive query requires approval", approval_required=True)

        if record.privacy == DocumentPrivacyClassification.OPEN_NOT_PUBLIC:
            return AccessDecision(AccessAction.REDACT, "non-public content is returned with redaction")

        if record.privacy == DocumentPrivacyClassification.PRIVATE:
            return AccessDecision(AccessAction.REDACT, "private content is returned with redaction")

        return AccessDecision(AccessAction.REQUIRE_APPROVAL, "sensitive content access requires approval", approval_required=True)

    def _is_targeted_sensitive_query(self, query: str) -> bool:
        return bool(_SENSITIVE_TEXT_RE.search(query))

    def redact_text(self, text: str) -> str:
        return redact_sensitive_text(text)

    def redact_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        redacted: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                redacted[key] = None
                continue
            value_text = str(value)
            if _SENSITIVE_TEXT_RE.search(key) or _SENSITIVE_TEXT_RE.search(value_text) or _PII_RE.search(value_text):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = value
        return redacted


class AccessAuditLogger:
    def __init__(self) -> None:
        self._logger = logging.getLogger("document_mgmt_service.audit")

    def record(self, event: AccessAuditEvent) -> None:
        payload = {
            "timestamp": event.timestamp.isoformat(),
            "document_id": event.request.document_id,
            "version": event.request.version,
            "intent": event.request.intent.value,
            "query": event.request.query,
            "requestor": event.request.requestor,
            "decision": event.decision.action.value,
            "reason": event.decision.reason,
            "privacy": event.document_privacy.value,
            "redacted": event.redacted,
        }
        level = logging.INFO if event.decision.action == AccessAction.ALLOW else logging.WARNING
        self._logger.log(level, "access_audit %s", json.dumps(payload, separators=(",", ":")))


@dataclass(frozen=True, slots=True)
class AccessResponse:
    decision: AccessDecision
    payload: dict[str, Any]


class DocumentAccessService:
    def __init__(
        self,
        *,
        repository: PostgreSQLDocumentRepository,
        storage: FileStorage,
        search: SemanticSearchService,
        policy: DocumentAccessPolicy | None = None,
        auditor: AccessAuditLogger | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._search = search
        self._policy = policy or DocumentAccessPolicy()
        self._auditor = auditor or AccessAuditLogger()

    @property
    def repository(self) -> PostgreSQLDocumentRepository:
        """Read-only access to the underlying document repository.

        Used by HTTP-layer tools that need to look up records without
        going through the full access-policy evaluation pipeline (e.g.
        ``get_field_value`` does its own policy check after the user has
        already approved the read). Public on purpose: callers in the
        adapter layer must not reach into ``_repository`` directly.
        """
        return self._repository

    def get_document_metadata(self, document_id: str, version: int) -> AccessResponse:
        record = self._load_record(document_id, version)
        request = AccessRequest(document_id=document_id, version=version, intent=AccessIntent.METADATA)
        decision = self._policy.evaluate(record=record, request=request)
        self._audit(record, request, decision)
        payload = self._base_record_payload(record)
        payload["access_action"] = decision.action.value
        payload["access_reason"] = decision.reason
        return AccessResponse(decision=decision, payload=payload)

    def get_document_description(self, document_id: str, version: int) -> AccessResponse:
        record = self._load_record(document_id, version)
        request = AccessRequest(document_id=document_id, version=version, intent=AccessIntent.DESCRIPTION)
        decision = self._policy.evaluate(record=record, request=request)
        self._audit(record, request, decision)
        payload = {
            "document_id": record.document_id,
            "version": record.version,
            "description": record.description,
            "access_action": decision.action.value,
            "access_reason": decision.reason,
        }
        return AccessResponse(decision=decision, payload=payload)

    def get_metadata(self, document_id: str, version: int) -> AccessResponse:
        return self.get_document_metadata(document_id, version)

    def list_documents(self) -> dict[str, Any]:
        documents = self._repository.list_documents()
        results = [self._summary_payload(summary) for summary in documents]
        return {"results": results}

    def search_documents(self, query: str, *, limit: int = 10, requestor: str | None = None) -> dict[str, Any]:
        hits = self._search.search_documents(query, limit=limit)
        results: list[dict[str, Any]] = []
        top_decision = AccessDecision(AccessAction.ALLOW, "document-level summaries are low-risk")
        for hit in hits:
            record = self._load_record(hit.document_id, hit.version)
            request = AccessRequest(
                document_id=hit.document_id,
                version=hit.version,
                intent=AccessIntent.DOCUMENT_SEARCH,
                query=query,
                limit=limit,
                requestor=requestor,
            )
            decision = self._policy.evaluate(record=record, request=request)
            self._audit(record, request, decision, redacted=decision.action == AccessAction.REDACT)
            result = self._document_hit_payload(hit)
            result["access_action"] = decision.action.value
            result["access_reason"] = decision.reason
            if decision.action == AccessAction.DENY:
                continue
            if decision.action == AccessAction.REQUIRE_APPROVAL:
                result["evidence"] = []
                result["description"] = record.description
                results.append(result)
                continue
            if decision.action == AccessAction.REDACT:
                result["description"] = self._policy.redact_text(result.get("description") or "")
                result["metadata"] = self._policy.redact_metadata(result.get("metadata") or {})
                result["evidence"] = [self._redact_evidence_item(item) for item in result.get("evidence", [])]
            results.append(result)
        if not results and hits:
            top_decision = AccessDecision(AccessAction.DENY, "no search results were permitted")
        return {
            "query": query,
            "access_action": top_decision.action.value,
            "access_reason": top_decision.reason,
            "results": results,
        }

    def search_document_content(self, query: str, *, limit: int = 10, document_id: str | None = None, version: int | None = None, requestor: str | None = None) -> dict[str, Any]:
        hits = self._search.search_content(query, limit=limit, document_id=document_id, version=version)
        results: list[dict[str, Any]] = []
        for hit in hits:
            record = self._load_record(hit.document_id, hit.version)
            request = AccessRequest(
                document_id=hit.document_id,
                version=hit.version,
                intent=AccessIntent.CONTENT_SEARCH,
                query=query,
                limit=limit,
                requestor=requestor,
            )
            decision = self._policy.evaluate(record=record, request=request)
            self._audit(record, request, decision, redacted=decision.action == AccessAction.REDACT)
            if decision.action == AccessAction.DENY:
                continue
            payload = self._content_hit_payload(hit)
            payload["access_action"] = decision.action.value
            payload["access_reason"] = decision.reason
            if decision.action == AccessAction.REQUIRE_APPROVAL:
                payload["text"] = ""
            elif decision.action == AccessAction.REDACT:
                payload["text"] = self._policy.redact_text(payload["text"])
                payload["metadata"] = self._policy.redact_metadata(payload.get("metadata") or {})
            results.append(payload)
        return {
            "query": query,
            "results": results,
        }

    def search_content(self, query: str, *, limit: int = 10, document_id: str | None = None, version: int | None = None, requestor: str | None = None) -> dict[str, Any]:
        return self.search_document_content(
            query,
            limit=limit,
            document_id=document_id,
            version=version,
            requestor=requestor,
        )

    def get_evidence(self, *, document_id: str, version: int, query: str, limit: int = 5, requestor: str | None = None) -> dict[str, Any]:
        record = self._load_record(document_id, version)
        request = AccessRequest(
            document_id=document_id,
            version=version,
            intent=AccessIntent.EVIDENCE,
            query=query,
            limit=limit,
            requestor=requestor,
        )
        decision = self._policy.evaluate(record=record, request=request)
        self._audit(record, request, decision, redacted=decision.action == AccessAction.REDACT)
        if decision.action in {AccessAction.DENY, AccessAction.REQUIRE_APPROVAL}:
            return {
                "query": query,
                "access_action": decision.action.value,
                "access_reason": decision.reason,
                "results": [],
            }
        hits = self._search.retrieve_evidence(document_id=document_id, version=version, query=query, limit=limit)
        results = [self._evidence_payload(hit, redacted=decision.action == AccessAction.REDACT) for hit in hits]
        return {
            "query": query,
            "access_action": decision.action.value,
            "access_reason": decision.reason,
            "results": results,
        }

    def retrieve_evidence(self, *, document_id: str, version: int, query: str, limit: int = 5, requestor: str | None = None) -> dict[str, Any]:
        return self.get_evidence(
            document_id=document_id,
            version=version,
            query=query,
            limit=limit,
            requestor=requestor,
        )

    def get_page(self, document_id: str, version: int, page_number: int, *, requestor: str | None = None) -> AccessResponse:
        record = self._load_record(document_id, version)
        request = AccessRequest(
            document_id=document_id,
            version=version,
            intent=AccessIntent.PAGE,
            query=f"page:{page_number}",
            requestor=requestor,
        )
        decision = self._policy.evaluate(record=record, request=request)
        self._audit(record, request, decision, redacted=decision.action == AccessAction.REDACT)
        pages = (record.extracted_text or "").split("\f") if record.extracted_text else []
        if page_number < 1 or page_number > max(len(pages), 1):
            raise KeyError(f"Page {page_number} not found for {document_id} v{version}")
        page_text = pages[page_number - 1] if pages else ""
        if decision.action == AccessAction.DENY:
            raise AccessDeniedError(decision)
        if decision.action == AccessAction.REQUIRE_APPROVAL:
            raise ApprovalRequiredError(decision)
        payload = {
            "document_id": record.document_id,
            "version": record.version,
            "page_number": page_number,
            "text": page_text if decision.action == AccessAction.ALLOW else self._policy.redact_text(page_text),
            "privacy": record.privacy.value,
            "access_action": decision.action.value,
            "access_reason": decision.reason,
        }
        return AccessResponse(decision=decision, payload=payload)

    def get_document(self, document_id: str, version: int, *, requestor: str | None = None) -> AccessResponse:
        record = self._load_record(document_id, version)
        request = AccessRequest(
            document_id=document_id,
            version=version,
            intent=AccessIntent.WHOLE_DOCUMENT,
            requestor=requestor,
        )
        decision = self._policy.evaluate(record=record, request=request)
        self._audit(record, request, decision)
        if decision.action == AccessAction.DENY:
            raise AccessDeniedError(decision)
        if decision.action == AccessAction.REQUIRE_APPROVAL:
            raise ApprovalRequiredError(decision)
        path = self._storage.get(record.storage_key)
        payload = {
            "document_id": record.document_id,
            "version": record.version,
            "filename": record.original_filename,
            "content_type": record.content_type or self._guess_content_type(path),
            "storage_key": record.storage_key,
            "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            "access_action": decision.action.value,
            "access_reason": decision.reason,
        }
        return AccessResponse(decision=decision, payload=payload)

    def retrieve_document(self, document_id: str, version: int, *, requestor: str | None = None) -> AccessResponse:
        return self.get_document(document_id, version, requestor=requestor)

    def request_sensitive_access(
        self,
        *,
        document_id: str,
        version: int,
        intent: AccessIntent,
        query: str | None = None,
        requestor: str | None = None,
    ) -> dict[str, Any]:
        record = self._load_record(document_id, version)
        request = AccessRequest(
            document_id=document_id,
            version=version,
            intent=intent,
            query=query,
            requestor=requestor,
        )
        decision = self._policy.evaluate(record=record, request=request)
        self._audit(record, request, decision, redacted=True)
        return {
            "document_id": document_id,
            "version": version,
            "intent": intent.value,
            "query": query,
            "access_action": decision.action.value,
            "access_reason": decision.reason,
            "approval_required": decision.approval_required,
        }

    def _load_record(self, document_id: str, version: int) -> DocumentVersionRecord:
        record = self._repository.get_version(document_id, version)
        if record is None:
            raise KeyError(f"Document {document_id} version {version} not found")
        return record

    def _audit(
        self,
        record: DocumentVersionRecord,
        request: AccessRequest,
        decision: AccessDecision,
        *,
        redacted: bool = False,
    ) -> None:
        if decision.action == AccessAction.ALLOW and record.privacy == DocumentPrivacyClassification.OPEN:
            return
        self._auditor.record(
            AccessAuditEvent(
                timestamp=datetime.now(timezone.utc),
                request=request,
                decision=decision,
                document_privacy=record.privacy,
                redacted=redacted,
            )
        )

    def _base_record_payload(self, record: DocumentVersionRecord) -> dict[str, Any]:
        return {
            "document_id": record.document_id,
            "version": record.version,
            "processing_status": record.processing_status.value,
            "privacy": record.privacy.value,
            "metadata": record.metadata,
            "description": record.description,
            "semantic_index_status": record.semantic_index_status.value,
            "chunk_count": record.chunk_count,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        }

    def _summary_payload(self, summary: DocumentSummaryRecord) -> dict[str, Any]:
        return {
            "document_id": summary.document_id,
            "latest_version": summary.latest_version,
            "processing_status": summary.processing_status.value,
            "privacy": summary.privacy.value,
            "metadata": dict(summary.metadata),
            "description": summary.description,
            "summary": summary.summary,
            "owner": {
                "type": summary.owner_type,
                "relation": summary.relation,
                "name": summary.relation_name,
            } if summary.owner_type else None,
            "expiry_date": summary.expiry_date.isoformat() if summary.expiry_date else None,
            "extracted_fields": list(summary.extracted_fields.keys()) if summary.extracted_fields else None,
            "semantic_index_status": summary.semantic_index_status.value,
            "chunk_count": summary.chunk_count,
            "created_at": summary.created_at.isoformat() if summary.created_at else None,
            "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
        }

    def _document_hit_payload(self, hit: SemanticDocumentSearchResult) -> dict[str, Any]:
        return {
            "document_id": hit.document_id,
            "version": hit.version,
            "score": hit.score,
            "privacy": hit.privacy.value,
            "description": hit.description,
            "metadata": dict(hit.metadata),
            "provenance": dict(hit.provenance),
            "evidence": [dict(item) for item in hit.evidence],
        }

    def _content_hit_payload(self, hit: SemanticContentSearchResult) -> dict[str, Any]:
        return {
            "chunk_id": hit.chunk_id,
            "document_id": hit.document_id,
            "version": hit.version,
            "score": hit.score,
            "privacy": hit.privacy.value,
            "text": hit.text,
            "metadata": dict(hit.metadata),
            "provenance": dict(hit.provenance),
        }

    def _evidence_payload(self, hit: SemanticEvidenceResult, *, redacted: bool) -> dict[str, Any]:
        text = self._policy.redact_text(hit.text) if redacted else hit.text
        metadata = self._policy.redact_metadata(hit.metadata) if redacted else dict(hit.metadata)
        return {
            "document_id": hit.document_id,
            "version": hit.version,
            "query": hit.query,
            "score": hit.score,
            "privacy": hit.privacy.value,
            "text": text,
            "metadata": metadata,
            "provenance": dict(hit.provenance),
        }

    def _redact_evidence_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        item["text"] = self._policy.redact_text(str(item.get("text") or ""))
        item["metadata"] = self._policy.redact_metadata(item.get("metadata") or {})
        return item

    def _guess_content_type(self, path: Path) -> str | None:
        guessed, _ = mimetypes.guess_type(str(path))
        return guessed
