from __future__ import annotations

import base64
import cgi
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from document_mgmt_service.application.ingestion import IngestionError, IngestionService
from document_mgmt_service.application.health import HealthService
from document_mgmt_service.application.access import (
    AccessDeniedError,
    ApprovalRequiredError,
    DocumentAccessService,
)
from document_mgmt_service.application.fields import (
    FieldAccessError,
    get_field_value,
)
from document_mgmt_service.domain.models import (
    DocumentIngestionRequest,
    DocumentPrivacyClassification,
)


_STATUS_PATH_RE = re.compile(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/status$")

# ---------------------------------------------------------------------------
# Confirmation-token machinery for the two-step get_field_value protocol.
# ---------------------------------------------------------------------------
_FIELD_CONFIRM_TTL_SECONDS = 300  # 5 minutes


class _BadRequest(ValueError):
    """Raised after a 400 response has already been sent.

    Helpers like :meth:`_required_field` send the error response
    themselves, then raise this to abort the handler. Callers must
    catch it and return WITHOUT sending again (a second send corrupts
    the HTTP stream). Subclasses ValueError so existing
    ``except ValueError`` guards keep working.
    """


_FIELD_CONFIRM_KEY_CACHE: bytes | None = None


def _get_field_confirm_key() -> bytes:
    """Resolve the HMAC key for signing confirmation tokens."""
    global _FIELD_CONFIRM_KEY_CACHE
    if _FIELD_CONFIRM_KEY_CACHE is not None:
        return _FIELD_CONFIRM_KEY_CACHE
    raw = os.environ.get("CHITRAGUPTA_FIELD_CONFIRM_KEY")
    if raw:
        key = bytes.fromhex(raw) if len(raw) % 2 == 0 else raw.encode("utf-8")
    else:
        env = os.environ.get("CHITRAGUPTA_ENV", "").lower()
        if env in {"prod", "production"}:
            raise RuntimeError(
                "CHITRAGUPTA_FIELD_CONFIRM_KEY must be set in production; "
                "refusing to start with a per-process random key."
            )
        key = secrets.token_bytes(32)
        logging.getLogger(__name__).warning(
            "CHITRAGUPTA_FIELD_CONFIRM_KEY is not set; using a per-process "
            "random key. Outstanding tokens will be invalidated on restart."
        )
    _FIELD_CONFIRM_KEY_CACHE = key
    return key


def _sign_field_confirm_token(
    *, document_id: str, version: int, field_name: str, ttl: int
) -> str:
    """Build a base64url HMAC-signed confirmation token."""
    expiry = int(time.time()) + ttl
    nonce = secrets.token_hex(8)
    payload = f"{expiry}:{nonce}:{document_id}:{version}:{field_name}"
    sig = hmac.new(_get_field_confirm_key(), payload.encode("utf-8"),
                   hashlib.sha256).hexdigest()
    raw = f"{payload}:{sig}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _verify_field_confirm_token(
    token: str, *, document_id: str, version: int, field_name: str
) -> bool:
    """Return True iff the token is valid for the given call."""
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError, base64.binascii.Error):
        return False
    # Reject non-canonical encodings: base64 quanta with pad bits admit
    # multiple spellings of the same bytes, so a mutated token can decode
    # identically and pass the HMAC check. Re-encoding must round-trip.
    canonical = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    if not hmac.compare_digest(canonical, token):
        return False
    parts = raw.split(":")
    if len(parts) != 6:
        return False
    expiry_str, nonce, tok_doc, tok_ver, tok_field, sig = parts
    try:
        expiry = int(expiry_str)
        tok_version = int(tok_ver)
    except ValueError:
        return False
    if expiry < int(time.time()):
        return False
    if tok_doc != document_id or tok_version != version or tok_field != field_name:
        return False
    payload = f"{expiry_str}:{nonce}:{tok_doc}:{tok_version}:{tok_field}"
    expected = hmac.new(
        _get_field_confirm_key(), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


def _json_safe(obj):
    """Recursive serializer that handles dataclasses and datetime."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class _DocumentRequestHandler(BaseHTTPRequestHandler):
    health_service: HealthService
    ingestion_service: IngestionService
    access_service: DocumentAccessService

    def _request_user_id(self, args: dict | None = None) -> str | None:
        """Tenant for this request: explicit arg wins, then X-User-Id header."""
        if args and args.get("user_id"):
            return str(args["user_id"])
        headers = getattr(self, "headers", None)
        header = (headers.get("X-User-Id") or "").strip() if headers else ""
        return header or None

    def _owned_record(self, document_id: str, version: int, user_id: str | None):
        """Load-or-404 with ownership enforcement (cross-user reads 404)."""
        try:
            return self.access_service._load_record(document_id, version, user_id=user_id)
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return None

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/healthz", "/readyz", "/"}:
            self._send_json(HTTPStatus.OK, self.health_service.report().to_dict())
            return

        # Generic tool dispatch: GET /tools/{tool_name}?args...
        # Used by the orchestrator's HttpServiceClient which expects a
        # /tools/{name} shape.
        tool_match = re.match(r"^/tools/(?P<tool_name>[^/]+)$", urlparse(self.path).path)
        if tool_match:
            self._handle_tool_dispatch(tool_match.group("tool_name"), query_params=True)
            return

        metadata_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/metadata$", urlparse(self.path).path)
        if metadata_match:
            if self._owned_record(metadata_match.group("document_id"), int(metadata_match.group("version")), self._request_user_id()) is None:
                return
            self._handle_metadata_retrieval(metadata_match.group("document_id"), int(metadata_match.group("version")))
            return

        match = _STATUS_PATH_RE.match(urlparse(self.path).path)
        if match:
            try:
                status = self.ingestion_service.get_status(
                    match.group("document_id"),
                    int(match.group("version")),
                )
            except IngestionError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "document_id": status.document_id,
                    "version": status.version,
                    "processing_status": status.processing_status.value,
                    "privacy": status.privacy.value,
                    "metadata": status.metadata,
                    "description": status.description,
                    "semantic_index_status": status.semantic_index_status.value,
                    "chunk_count": status.chunk_count,
                    "error_message": status.error_message,
                    "created_at": status.created_at.isoformat() if status.created_at else None,
                    "updated_at": status.updated_at.isoformat() if status.updated_at else None,
                    "completed_at": status.completed_at.isoformat() if status.completed_at else None,
                },
            )
            return

        ocr_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/ocr$", urlparse(self.path).path)
        if ocr_match:
            if self._owned_record(ocr_match.group("document_id"), int(ocr_match.group("version")), self._request_user_id()) is None:
                return
            self._handle_ocr_retrieval(ocr_match.group("document_id"), int(ocr_match.group("version")))
            return

        image_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/image$", urlparse(self.path).path)
        if image_match:
            if self._owned_record(image_match.group("document_id"), int(image_match.group("version")), self._request_user_id()) is None:
                return
            self._handle_image_retrieval(image_match.group("document_id"), int(image_match.group("version")))
            return

        file_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/file$", urlparse(self.path).path)
        if file_match:
            _fuid = self._request_user_id()
            try:
                response = self.access_service.retrieve_document(
                    file_match.group("document_id"),
                    int(file_match.group("version")),
                    user_id=_fuid,
                )
            except ApprovalRequiredError as exc:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": str(exc), "access_action": exc.decision.action.value, "access_reason": exc.decision.reason},
                )
                return
            except AccessDeniedError as exc:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": str(exc), "access_action": exc.decision.action.value, "access_reason": exc.decision.reason},
                )
                return
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, response.payload)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/documents":
                self._handle_upload()
                return

            if path == "/documents/restore":
                self._handle_restore()
                return

            if path == "/search/documents":
                self._handle_document_search()
                return

            if path == "/search/content":
                self._handle_content_search()
                return

            if path == "/alerts":
                self._handle_alert()
                return

            # Generic tool dispatch: POST /tools/{tool_name}
            # The orchestrator's HttpServiceClient posts to this path.
            tool_match = re.match(r"^/tools/(?P<tool_name>[^/]+)$", path)
            if tool_match:
                self._handle_tool_dispatch(tool_match.group("tool_name"), query_params=False)
                return

            evidence_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/evidence$", path)
            if evidence_match:
                self._handle_evidence_search(evidence_match.group("document_id"), int(evidence_match.group("version")))
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except IngestionError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive server guard
            logging.getLogger("document_mgmt_service.http").exception("Request failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_restore(self) -> None:
        """V2-4 version-preserving restore for device pull.

        Body: {document_id, version, user_id, filename, content_type,
        content_base64, metadata?}. Re-inserts the exact version (higher
        version wins; stale restores are acknowledged without overwrite)
        so a wiped device pulls back identical IDs/versions.
        """
        from document_mgmt_service.application.ingestion import _lock_for_document
        from document_mgmt_service.domain.models import (
            DocumentPrivacyClassification, DocumentProcessingStatus,
        )
        payload = self._read_json_body()
        document_id = str(payload.get("document_id") or "")
        version = int(payload.get("version") or 0)
        user_id = str(payload.get("user_id") or self.headers.get("X-User-Id") or "__local__")
        if not document_id or version <= 0:
            raise IngestionError("document_id and positive version are required")
        try:
            content = base64.b64decode(payload.get("content_base64") or "")
        except Exception as exc:
            raise IngestionError(f"invalid content_base64: {exc}")
        if not content:
            raise IngestionError("empty restore content")
        svc = self.ingestion_service
        deps = svc._dependencies
        lock = _lock_for_document(document_id)
        lock.acquire()
        try:
            existing = deps.repository.get_version(document_id, version)
            if existing is not None and getattr(existing, "user_id", "__local__") == user_id:
                self._send_json(HTTPStatus.OK, {"restored": False, "reason": "already present",
                                                "document_id": document_id, "version": version})
                return
            latest = deps.repository.list_versions(document_id)
            if any(v.version > version for v in latest):
                self._send_json(HTTPStatus.OK, {"restored": False, "reason": "newer version present",
                                                "document_id": document_id, "version": version})
                return
            import tempfile
            from hashlib import sha256 as _sha
            from pathlib import Path as _Path
            from document_mgmt_service.application.metadata import infer_file_kind
            filename = str(payload.get("filename") or f"{document_id}_v{version}")
            file_kind = infer_file_kind(filename, payload.get("content_type"))
            storage_key = svc._storage_key(document_id, version, filename)
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(content)
                tmp = _Path(tf.name)
            try:
                deps.storage.put(tmp, storage_key)
            finally:
                tmp.unlink(missing_ok=True)
            now = svc._clock()
            from document_mgmt_service.domain.models import (
                DocumentVersionRecord, SemanticIndexStatus,
            )
            record = DocumentVersionRecord(
                document_id=document_id, version=version, user_id=user_id,
                original_filename=filename, content_type=payload.get("content_type"),
                file_kind=file_kind, storage_key=storage_key,
                file_size_bytes=len(content), sha256=_sha(content).hexdigest(),
                privacy=DocumentPrivacyClassification.OPEN_NOT_PUBLIC,
                processing_status=DocumentProcessingStatus.COMPLETED,
                metadata=dict(payload.get("metadata") or {}),
                semantic_index_status=SemanticIndexStatus.PENDING,
                created_at=now, updated_at=now, completed_at=now,
            )
            deps.repository.upsert_version(record)
            try:
                if deps.semantic_search is not None:
                    deps.semantic_search.index_document(record)
            except Exception:
                logging.getLogger("document_mgmt_service.http").exception("Restore reindex failed")
            self._send_json(HTTPStatus.CREATED, {"restored": True, "document_id": document_id,
                                                 "version": version, "user_id": user_id})
        finally:
            lock.release()

    def _handle_upload(self) -> None:
        try:
            request = self._parse_ingestion_request()
            _uid = (self.headers.get("X-User-Id") or "").strip() or "__local__"
            if getattr(request, "user_id", "__local__") in (None, "", "__local__") and _uid != "__local__":
                import dataclasses
                request = dataclasses.replace(request, user_id=_uid)

            # Check for multi-page PDF
            from document_mgmt_service.infrastructure.pdf_splitter import (
                is_pdf, split_pdf,
            )

            if is_pdf(request.content_type, request.original_filename):
                split_result = split_pdf(request.content)

                if split_result.is_multi_page and split_result.pages:
                    # Ingest each page as a separate document
                    documents = []
                    for page in split_result.pages:
                        # Convert page image to bytes with PDF-like naming
                        page_filename = f"{request.original_filename}_page{page.page_number}.png"
                        page_request = DocumentIngestionRequest(
                            original_filename=page_filename,
                            content=page.image_bytes,
                            content_type=page.content_type,
                            document_id=None,  # New document_id per page
                            user_id=getattr(request, "user_id", "__local__") or "__local__",
                            privacy_hint=request.privacy_hint,
                            description_hint=f"Page {page.page_number} of {split_result.total_pages}",
                            metadata={
                                **dict(request.metadata),
                                "source_pdf": request.original_filename,
                                "page_number": page.page_number,
                                "total_pages": split_result.total_pages,
                                "page_width": page.width,
                                "page_height": page.height,
                            },
                        )
                        result = self.ingestion_service.ingest(page_request)
                        documents.append({
                            "status": "successful",
                            "document_id": result.document_id,
                            "version": result.version,
                            "processing_status": result.processing_status.value,
                            "page_number": page.page_number,
                            "total_pages": split_result.total_pages,
                            "status_url": f"/documents/{result.document_id}/versions/{result.version}/status",
                        })

                    self._send_json(
                        HTTPStatus.CREATED,
                        {
                            "status": "successful",
                            "documents": documents,
                            "total_pages": split_result.total_pages,
                            "message": f"Uploaded successfully. {len(documents)} page documents created.",
                        },
                    )
                    return

            # Single document (image or single-page PDF)
            result = self.ingestion_service.ingest(request)
            self._send_json(
                HTTPStatus.CREATED,
                {
                    "status": "successful",
                    "message": "Document uploaded successfully.",
                    "document_id": result.document_id,
                    "version": result.version,
                    "processing_status": result.processing_status.value,
                    "semantic_index_status": result.semantic_index_status.value,
                    "chunk_count": result.chunk_count,
                    "status_url": f"/documents/{result.document_id}/versions/{result.version}/status",
                },
            )
        except IngestionError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive server guard
            logging.getLogger("document_mgmt_service.http").exception("Upload failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_document_search(self) -> None:
        payload = self._read_json_body()
        query = self._required_query(payload)
        results = self.access_service.search_documents(
            query,
            limit=int(payload.get("limit", 10)),
            requestor=self._first_context(payload),
        )
        self._send_json(HTTPStatus.OK, results)

    def _handle_content_search(self) -> None:
        payload = self._read_json_body()
        query = self._required_query(payload)
        results = self.access_service.search_content(
            query,
            limit=int(payload.get("limit", 10)),
            document_id=payload.get("document_id"),
            version=int(payload["version"]) if payload.get("version") is not None else None,
            requestor=self._first_context(payload),
        )
        self._send_json(HTTPStatus.OK, results)

    def _handle_evidence_search(self, document_id: str, version: int) -> None:
        payload = self._read_json_body()
        query = self._required_query(payload)
        results = self.access_service.retrieve_evidence(
            document_id=document_id,
            version=version,
            query=query,
            limit=int(payload.get("limit", 5)),
            requestor=self._first_context(payload),
        )
        self._send_json(HTTPStatus.OK, results)

    def _handle_alert(self) -> None:
        """Handle validation alerts from the validator service.

        Alerts are logged and optionally used to update document status.
        """
        payload = self._read_json_body()
        document_id = payload.get("document_id")
        version = payload.get("version")
        severity = payload.get("severity", "INFO")
        alert_type = payload.get("alert_type", "unknown")
        message = payload.get("message", "")
        risk_score = payload.get("risk_score", 0.0)
        source = payload.get("source", "unknown")

        if not document_id:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "document_id is required"})
            return

        logging.getLogger("document_mgmt_service.http").warning(
            "Validation alert received: document=%s v%s severity=%s type=%s risk=%.2f source=%s message=%s",
            document_id, version, severity, alert_type, risk_score, source, message,
        )

        # For CRITICAL alerts, update document metadata to flag it
        if severity == "CRITICAL" and version is not None:
            try:
                record = self.access_service._load_record(document_id, int(version))
                updated_metadata = dict(record.metadata)
                updated_metadata["validation_alert"] = {
                    "severity": severity,
                    "alert_type": alert_type,
                    "message": message,
                    "risk_score": risk_score,
                    "source": source,
                    "timestamp": payload.get("timestamp"),
                }
                updated_metadata["validation_status"] = "INVALID"
                # Update the record with the flagged metadata
                from document_mgmt_service.domain.models import DocumentVersionRecord
                flagged_record = DocumentVersionRecord(
                    document_id=record.document_id,
                    version=record.version,
                    user_id=getattr(record, "user_id", "__local__") or "__local__",
                    original_filename=record.original_filename,
                    content_type=record.content_type,
                    file_kind=record.file_kind,
                    storage_key=record.storage_key,
                    file_size_bytes=record.file_size_bytes,
                    sha256=record.sha256,
                    privacy=record.privacy,
                    processing_status=record.processing_status,
                    metadata=updated_metadata,
                    description=record.description,
                    extracted_text=record.extracted_text,
                    extracted_text_excerpt=record.extracted_text_excerpt,
                    semantic_index_status=record.semantic_index_status,
                    semantic_indexed_at=record.semantic_indexed_at,
                    chunk_count=record.chunk_count,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    completed_at=record.completed_at,
                    error_message=f"VALIDATION_ALERT: {message}",
                    model_extraction=record.model_extraction,
                    description_safe=record.description_safe,
                    description_detailed=record.description_detailed,
                    extraction_confidence=record.extraction_confidence,
                    document_type=record.document_type,
                    document_sub_type=record.document_sub_type,
                    language_primary=record.language_primary,
                    pii_types=record.pii_types,
                )
                self.ingestion_service._repo.upsert_version(flagged_record)
                logging.getLogger("document_mgmt_service.http").info(
                    "Document %s v%s flagged as INVALID by validator service",
                    document_id, version,
                )
            except Exception as exc:
                logging.getLogger("document_mgmt_service.http").warning(
                    "Could not flag document %s: %s", document_id, exc,
                )

        self._send_json(HTTPStatus.ACCEPTED, {
            "alert_id": f"alert-{document_id}-{version}-{alert_type}",
            "status": "received",
            "document_id": document_id,
            "version": version,
        })

    # ------------------------------------------------------------------
    # Generic /tools/{name} dispatch
    # ------------------------------------------------------------------
    # The orchestrator's HttpServiceClient calls POST /tools/{tool_name} for
    # every tool, but this HTTP server exposes specific routes like
    # /search/documents and /documents. This dispatch table maps tool names
    # to handler methods so the orchestrator's calls succeed without the
    # orchestrator needing to know each route.
    _TOOL_DISPATCH_GET = {
        "list_documents": "_tool_list_documents",
        "list_expiring_documents": "_tool_list_expiring_documents",
        "list_templates": "_tool_list_templates",
        "get_template": "_tool_get_template",
        "get_alerts": "_tool_get_alerts",
        "get_document": "_tool_get_document",
        "get_document_ocr": "_tool_get_document_ocr",
        "get_page": "_tool_get_page",
        "get_document_metadata": "_tool_get_document_metadata",
        "get_document_description": "_tool_get_document_description",
    }
    _TOOL_DISPATCH_POST = {
        "list_documents": "_tool_list_documents",
        "list_expiring_documents": "_tool_list_expiring_documents",
        "search_documents": "_tool_search_documents",
        "search_document_content": "_tool_search_content",
        "get_evidence": "_tool_get_evidence",
        "get_field_value": "_tool_get_field_value",
        "upload_document": "_tool_upload_document",
    }

    def _handle_tool_dispatch(self, tool_name: str, *, query_params: bool) -> None:
        """Dispatch a /tools/{name} request to the appropriate handler."""
        if query_params:
            table = self._TOOL_DISPATCH_GET
            try:
                args = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(urlparse(self.path).query).items()}
            except Exception:
                args = {}
        else:
            table = self._TOOL_DISPATCH_POST
            try:
                args = self._read_json_body()
            except Exception:
                args = {}

        method_name = table.get(tool_name)
        if method_name is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Tool '{tool_name}' is not exposed via /tools/ on this service"},
            )
            return

        handler = getattr(self, method_name, None)
        if handler is None:
            self._send_json(
                HTTPStatus.NOT_IMPLEMENTED,
                {"error": f"Tool '{tool_name}' is registered but its handler is not yet implemented"},
            )
            return

        try:
            handler(args)
        except IngestionError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("document_mgmt_service.http").exception(
                "Tool %s dispatch failed", tool_name,
            )
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    # --- /tools/{name} handler methods ---------------------------------

    def _tool_list_documents(self, args: dict) -> None:
        user_id = args.get("user_id") or self.headers.get("X-User-Id") or None
        results = self.access_service.list_documents(user_id=user_id)
        self._send_json(HTTPStatus.OK, results)

    def _tool_list_expiring_documents(self, args: dict) -> None:
        try:
            within_days = int(args.get("within_days", 60))
        except (TypeError, ValueError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "within_days must be an integer"},
            )
            return
        within_days = min(365, max(1, within_days))
        results = self.access_service.list_expiring_documents(within_days=within_days, user_id=self._request_user_id(args))
        self._send_json(HTTPStatus.OK, results)

    def _tool_search_documents(self, args: dict) -> None:
        query = self._required_query(args)
        results = self.access_service.search_documents(
            query,
            limit=int(args.get("limit", 10)),
            requestor=self._first_context(args),
            user_id=self._request_user_id(args),
        )
        self._send_json(HTTPStatus.OK, results)

    def _tool_search_content(self, args: dict) -> None:
        query = self._required_query(args)
        results = self.access_service.search_content(
            query,
            limit=int(args.get("limit", 10)),
            document_id=args.get("document_id"),
            version=int(args["version"]) if args.get("version") is not None else None,
            requestor=self._first_context(args),
            user_id=self._request_user_id(args),
        )
        self._send_json(HTTPStatus.OK, results)

    def _tool_get_evidence(self, args: dict) -> None:
        document_id = self._required_field(args, "document_id")
        version = int(self._required_field(args, "version"))
        query = self._required_query(args)
        results = self.access_service.retrieve_evidence(
            document_id=document_id, version=version, query=query,
            limit=int(args.get("limit", 5)),
            requestor=self._first_context(args),
            user_id=self._request_user_id(args),
        )
        self._send_json(HTTPStatus.OK, results)

    def _tool_get_field_value(self, args: dict) -> None:
        """Look up a single extracted field on a document version.

        Two-step protocol (policy §5, failures #1) — server-enforced:

          1. First call (no ``confirmation_token``) → status=
             "requires_confirmation" + a fresh HMAC-signed token bound to
             (document_id, version, field). The orchestrator/UI shows the
             confirmation popup and, after the user approves, re-issues the
             call with the token.
          2. Second call (with a valid, non-expired ``confirmation_token``
             whose embedded document_id / version / field match the args)
             → status="ok" and the actual value is returned.

        The token is server-issued; the client cannot synthesize one. This
        prevents the previous design where any caller could pass
        ``confirm=true`` directly and bypass the gate.
        """
        try:
            document_id = self._required_field(args, "document_id")
            version_raw = self._required_field(args, "version")
            field = self._required_field(args, "field")
        except _BadRequest:
            return  # 400 already sent by _required_field
        try:
            version = int(version_raw)
        except (TypeError, ValueError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "version must be an integer"},
            )
            return
        confirmation_token = args.get("confirmation_token") or ""

        # If the caller didn't present a valid token, treat the call as the
        # first step of the two-step protocol: ALWAYS return
        # requires_confirmation regardless of any client-side confirm flag.
        # (The legacy ``confirm`` arg is ignored for safety.)
        if not confirmation_token or not _verify_field_confirm_token(
            confirmation_token,
            document_id=document_id,
            version=version,
            field_name=field,
        ):
            try:
                preview = get_field_value(
                    self.access_service.repository,
                    document_id=document_id,
                    version=version,
                    field_name=field,
                    confirm=False,
                    user_id=self._request_user_id(args),
                )
            except FieldAccessError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            payload = _json_safe(preview)
            # Only a resolvable field gets a token: there is nothing to
            # confirm for a miss, and minting one invites confused retries.
            if preview.status == "requires_confirmation":
                payload["confirmation_token"] = _sign_field_confirm_token(
                    document_id=document_id,
                    version=version,
                    field_name=field,
                    ttl=_FIELD_CONFIRM_TTL_SECONDS,
                )
                payload["confirmation_ttl_seconds"] = _FIELD_CONFIRM_TTL_SECONDS
            self._send_json(HTTPStatus.OK, payload)
            return

        # Token valid — reveal the value.
        try:
            result = get_field_value(
                self.access_service.repository,
                document_id=document_id,
                version=version,
                field_name=field,
                confirm=True,
                user_id=self._request_user_id(args),
            )
        except FieldAccessError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, _json_safe(result))

    def _tool_upload_document(self, args: dict) -> None:
        """Ingest via the tools endpoint (parity with the MCP transport).

        Accepts the registry schema: filename, content_base64 (+ optional
        content_type, document_id, description, privacy, metadata).
        IngestionError propagates to the dispatcher, which maps it to 400.
        """
        content_base64 = args.get("content_base64") or ""
        filename = args.get("filename") or ""
        if not filename or not content_base64:
            raise IngestionError("filename and content_base64 are required")
        try:
            content = base64.b64decode(content_base64)
        except (ValueError, base64.binascii.Error) as exc:
            raise IngestionError(f"content_base64 is not valid base64: {exc}") from exc
        request = self._request_from_payload(dict(args), content=content)
        result = self.ingestion_service.ingest(request)
        self._send_json(HTTPStatus.OK, {
            "status": "successful",
            "message": "Document uploaded successfully.",
            "document_id": result.document_id,
            "version": result.version,
            "processing_status": result.processing_status.value,
            "semantic_index_status": result.semantic_index_status.value,
            "chunk_count": result.chunk_count,
        })

    def _tool_list_templates(self, args: dict) -> None:
        self._send_json(HTTPStatus.NOT_IMPLEMENTED, {"error": "list_templates not implemented on this service"})

    def _tool_get_template(self, args: dict) -> None:
        self._send_json(HTTPStatus.NOT_IMPLEMENTED, {"error": "get_template not implemented on this service"})

    def _tool_get_alerts(self, args: dict) -> None:
        self._send_json(HTTPStatus.OK, {"alerts": []})

    def _tool_get_document(self, args: dict) -> None:
        document_id = self._required_field(args, "document_id")
        version = int(self._required_field(args, "version"))
        try:
            response = self.access_service.retrieve_document(document_id, version, user_id=self._request_user_id(args))
        except (ApprovalRequiredError, AccessDeniedError) as exc:
            self._send_json(HTTPStatus.FORBIDDEN, {
                "error": str(exc),
                "access_action": exc.decision.action.value,
                "access_reason": exc.decision.reason,
            })
            return
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, response.payload)

    def _tool_get_document_ocr(self, args: dict) -> None:
        document_id = self._required_field(args, "document_id")
        version = int(self._required_field(args, "version"))
        try:
            record = self.access_service._load_record(document_id, version, user_id=self._request_user_id(args))
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, {"text": record.extracted_text or ""})

    def _tool_get_page(self, args: dict) -> None:
        document_id = self._required_field(args, "document_id")
        version = int(self._required_field(args, "version"))
        try:
            page_number = int(args.get("page_number", 1))
        except (TypeError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "page_number must be an integer"})
            return
        try:
            response = self.access_service.get_page(
                document_id, version, page_number, user_id=self._request_user_id(args)
            )
        except (ApprovalRequiredError, AccessDeniedError) as exc:
            self._send_json(HTTPStatus.FORBIDDEN, {
                "error": str(exc),
                "access_action": exc.decision.action.value,
                "access_reason": exc.decision.reason,
            })
            return
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, response.payload)

    def _tool_get_document_metadata(self, args: dict) -> None:
        document_id = self._required_field(args, "document_id")
        version = int(self._required_field(args, "version"))
        try:
            payload = self.access_service.get_document_metadata(document_id, version, user_id=self._request_user_id(args))
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, payload)

    def _tool_get_document_description(self, args: dict) -> None:
        document_id = self._required_field(args, "document_id")
        version = int(self._required_field(args, "version"))
        try:
            payload = self.access_service.get_document_description(document_id, version, user_id=self._request_user_id(args))
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, payload)

    def _required_query(self, payload: dict) -> str:
        q = payload.get("query")
        if not q or not str(q).strip():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "query is required"})
            raise _BadRequest("query required")
        return str(q)

    def _required_field(self, payload: dict, name: str):
        v = payload.get(name)
        if v is None or v == "":
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"{name} is required"})
            raise _BadRequest(f"{name} required")
        return v

    def _first_context(self, payload: dict) -> str | None:
        """Best-effort requestor extraction for access policy."""
        ctx = payload.get("context") or payload.get("requestor")
        if isinstance(ctx, dict):
            return ctx.get("user_id") or ctx.get("user")
        return ctx

    def _handle_metadata_retrieval(self, document_id: str, version: int) -> None:
        """Return full document metadata including model extraction fields."""
        try:
            record = self.access_service._load_record(document_id, version)
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "document_id": record.document_id,
                "version": record.version,
                "original_filename": record.original_filename,
                "content_type": record.content_type,
                "file_kind": record.file_kind.value,
                "processing_status": record.processing_status.value,
                "privacy": record.privacy.value,
                "metadata": record.metadata,
                "description": record.description,
                "extracted_text": record.extracted_text,
                "model_extraction": record.model_extraction,
                "document_type": record.document_type,
                "document_sub_type": record.document_sub_type,
                "language_primary": record.language_primary,
                "pii_types": record.pii_types,
                "extraction_confidence": record.extraction_confidence,
                "description_safe": record.description_safe,
                "description_detailed": record.description_detailed,
            },
        )

    def _handle_ocr_retrieval(self, document_id: str, version: int) -> None:
        """Return OCR text and metadata for a document version."""
        try:
            record = self._load_version_record(document_id, version)
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "document_id": record.document_id,
                "version": record.version,
                "processing_status": record.processing_status.value,
                "extracted_text": record.extracted_text,
                "extracted_text_excerpt": record.extracted_text_excerpt,
                "file_kind": record.file_kind.value,
                "original_filename": record.original_filename,
                "content_type": record.content_type,
                "metadata": record.metadata,
                "description": record.description,
                "privacy": record.privacy.value,
            },
        )

    def _handle_image_retrieval(self, document_id: str, version: int) -> None:
        """Return raw image/document bytes for a document version."""
        try:
            response = self.access_service.get_document(document_id, version)
        except ApprovalRequiredError as exc:
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": str(exc), "access_action": exc.decision.action.value, "access_reason": exc.decision.reason},
            )
            return
        except AccessDeniedError as exc:
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": str(exc), "access_action": exc.decision.action.value, "access_reason": exc.decision.reason},
            )
            return
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        content_b64 = response.payload.get("content_base64")
        content_type = response.payload.get("content_type", "application/octet-stream")
        if content_b64 is None:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Document access requires approval"})
            return
        raw_bytes = base64.b64decode(content_b64)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw_bytes)))
        self.end_headers()
        self.wfile.write(raw_bytes)

    def _load_version_record(self, document_id: str, version: int):
        """Load a version record from the repository via ingestion service."""
        from document_mgmt_service.domain.models import DocumentVersionRecord
        status = self.ingestion_service.get_status(document_id, version)
        # We need the full record for OCR fields — fetch from repository
        # The ingestion_service.get_status returns a status response,
        # but we need extracted_text which is on the full record.
        # Access via the repository through the access service's internal method.
        return self.access_service._load_record(document_id, version)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        logging.getLogger("document_mgmt_service.http").info(format, *args)

    def _parse_ingestion_request(self) -> DocumentIngestionRequest:
        content_type = self.headers.get_content_type()
        if content_type == "application/json":
            raw = self._read_body()
            payload = json.loads(raw.decode("utf-8"))
            content = base64.b64decode(payload["content_base64"])
            return self._request_from_payload(payload, content=content)

        if content_type.startswith("multipart/"):
            return self._parse_multipart_request()

        raw = self._read_body()
        if not raw:
            raise IngestionError("Missing upload body")
        query = parse_qs(urlparse(self.path).query)
        filename = self._first(query, "filename") or self.headers.get("X-Filename")
        if not filename:
            raise IngestionError("filename is required for raw uploads")
        return self._request_from_payload(
            {
                "filename": filename,
                "document_id": self._first(query, "document_id"),
                "privacy": self._first(query, "privacy"),
                "description": self._first(query, "description"),
                "metadata": self._first(query, "metadata"),
                "content_type": self.headers.get("Content-Type"),
            },
            content=raw,
        )

    def _parse_multipart_request(self) -> DocumentIngestionRequest:
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
        file_field = form["file"] if "file" in form else None
        if file_field is None or not getattr(file_field, "file", None):
            raise IngestionError("multipart upload requires a file field named 'file'")
        content = file_field.file.read()
        filename = file_field.filename or self._form_value(form, "filename")
        if not filename:
            raise IngestionError("Uploaded file name is required")
        payload = {
            "filename": filename,
            "document_id": self._form_value(form, "document_id"),
            "privacy": self._form_value(form, "privacy"),
            "description": self._form_value(form, "description"),
            "metadata": self._form_value(form, "metadata"),
            "content_type": file_field.type,
            "user_id": self._form_value(form, "user_id") or (self.headers.get("X-User-Id") or "").strip() or None,
        }
        return self._request_from_payload(payload, content=content)

    def _request_from_payload(self, payload: dict[str, Any], *, content: bytes) -> DocumentIngestionRequest:
        metadata: dict[str, Any] = {}
        raw_metadata = payload.get("metadata")
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as exc:
                raise IngestionError(f"Invalid metadata JSON: {exc}") from exc
        elif isinstance(raw_metadata, dict):
            metadata = raw_metadata

        privacy_value = payload.get("privacy")
        privacy = None
        if privacy_value:
            try:
                privacy = DocumentPrivacyClassification(str(privacy_value))
            except ValueError as exc:
                raise IngestionError(f"Invalid privacy classification: {privacy_value}") from exc

        filename = payload.get("filename")
        if not filename:
            raise IngestionError("filename is required")
        return DocumentIngestionRequest(
            original_filename=str(filename),
            content=content,
            content_type=payload.get("content_type"),
            document_id=payload.get("document_id"),
            privacy_hint=privacy,
            description_hint=payload.get("description"),
            metadata=metadata,
            user_id=str(payload.get("user_id") or metadata.get("user_id") or "__local__"),
        )

    def _read_json_body(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _required_query(self, payload: dict[str, Any]) -> str:
        query = payload.get("query")
        if not query:
            raise IngestionError("query is required")
        return str(query)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length > 0 else b""

    def _first(self, query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key)
        return values[0] if values else None

    def _form_value(self, form: cgi.FieldStorage, key: str) -> str | None:
        if key not in form:
            return None
        value = form.getfirst(key)
        return value if value not in {"", None} else None

    def _first_context(self, payload: dict[str, Any]) -> str | None:
        requestor = payload.get("requestor")
        return str(requestor) if requestor else None


class DocumentManagementHTTPServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        health_service: HealthService,
        ingestion_service: IngestionService,
        access_service: DocumentAccessService,
    ) -> None:
        handler = type(
            "DocumentRequestHandler",
            (_DocumentRequestHandler,),
            {
                "health_service": health_service,
                "ingestion_service": ingestion_service,
                "access_service": access_service,
            },
        )
        self._server = ThreadingHTTPServer((host, port), handler)

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.5)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
