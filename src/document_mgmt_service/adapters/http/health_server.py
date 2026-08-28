from __future__ import annotations

import base64
import cgi
import json
import logging
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
from document_mgmt_service.domain.models import (
    DocumentIngestionRequest,
    DocumentPrivacyClassification,
)


_STATUS_PATH_RE = re.compile(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/status$")


class _DocumentRequestHandler(BaseHTTPRequestHandler):
    health_service: HealthService
    ingestion_service: IngestionService
    access_service: DocumentAccessService

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/healthz", "/readyz", "/"}:
            self._send_json(HTTPStatus.OK, self.health_service.report().to_dict())
            return

        metadata_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/metadata$", urlparse(self.path).path)
        if metadata_match:
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
            self._handle_ocr_retrieval(ocr_match.group("document_id"), int(ocr_match.group("version")))
            return

        image_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/image$", urlparse(self.path).path)
        if image_match:
            self._handle_image_retrieval(image_match.group("document_id"), int(image_match.group("version")))
            return

        file_match = re.match(r"^/documents/(?P<document_id>[^/]+)/versions/(?P<version>\d+)/file$", urlparse(self.path).path)
        if file_match:
            try:
                response = self.access_service.retrieve_document(
                    file_match.group("document_id"),
                    int(file_match.group("version")),
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

            if path == "/search/documents":
                self._handle_document_search()
                return

            if path == "/search/content":
                self._handle_content_search()
                return

            if path == "/alerts":
                self._handle_alert()
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

    def _handle_upload(self) -> None:
        try:
            request = self._parse_ingestion_request()

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
